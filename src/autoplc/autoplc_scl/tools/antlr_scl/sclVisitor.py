# Generated from scl.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .sclParser import sclParser
else:
    from sclParser import sclParser

# This class defines a complete generic visitor for a parse tree produced by sclParser.

class sclVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by sclParser#r.
    def visitR(self, ctx:sclParser.RContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#whileStatement.
    def visitWhileStatement(self, ctx:sclParser.WhileStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#forStatement.
    def visitForStatement(self, ctx:sclParser.ForStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#ifStatement.
    def visitIfStatement(self, ctx:sclParser.IfStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#whileCondition.
    def visitWhileCondition(self, ctx:sclParser.WhileConditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#whileBlockStatements.
    def visitWhileBlockStatements(self, ctx:sclParser.WhileBlockStatementsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#repeatStatement.
    def visitRepeatStatement(self, ctx:sclParser.RepeatStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#switchStatement.
    def visitSwitchStatement(self, ctx:sclParser.SwitchStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#block.
    def visitBlock(self, ctx:sclParser.BlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#dataBlock.
    def visitDataBlock(self, ctx:sclParser.DataBlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#typeBlock.
    def visitTypeBlock(self, ctx:sclParser.TypeBlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#fbBlock.
    def visitFbBlock(self, ctx:sclParser.FbBlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#fcBlock.
    def visitFcBlock(self, ctx:sclParser.FcBlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#blockName.
    def visitBlockName(self, ctx:sclParser.BlockNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#dbInstanceName.
    def visitDbInstanceName(self, ctx:sclParser.DbInstanceNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#structType.
    def visitStructType(self, ctx:sclParser.StructTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#variableDefinitions.
    def visitVariableDefinitions(self, ctx:sclParser.VariableDefinitionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#variableDefinition.
    def visitVariableDefinition(self, ctx:sclParser.VariableDefinitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#weirdThing.
    def visitWeirdThing(self, ctx:sclParser.WeirdThingContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#constant_assign.
    def visitConstant_assign(self, ctx:sclParser.Constant_assignContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#variableType.
    def visitVariableType(self, ctx:sclParser.VariableTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#udtType.
    def visitUdtType(self, ctx:sclParser.UdtTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#elementaryType.
    def visitElementaryType(self, ctx:sclParser.ElementaryTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#arrayType.
    def visitArrayType(self, ctx:sclParser.ArrayTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#arrayRange.
    def visitArrayRange(self, ctx:sclParser.ArrayRangeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#dbAssignments.
    def visitDbAssignments(self, ctx:sclParser.DbAssignmentsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#dbAssignment.
    def visitDbAssignment(self, ctx:sclParser.DbAssignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#nonRetainBlock.
    def visitNonRetainBlock(self, ctx:sclParser.NonRetainBlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#blockVarDeclarations.
    def visitBlockVarDeclarations(self, ctx:sclParser.BlockVarDeclarationsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#blockTempVars.
    def visitBlockTempVars(self, ctx:sclParser.BlockTempVarsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#blockInOutDeclarations.
    def visitBlockInOutDeclarations(self, ctx:sclParser.BlockInOutDeclarationsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#blockInputDeclarations.
    def visitBlockInputDeclarations(self, ctx:sclParser.BlockInputDeclarationsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#blockOutputDeclarations.
    def visitBlockOutputDeclarations(self, ctx:sclParser.BlockOutputDeclarationsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#blockConstDeclarations.
    def visitBlockConstDeclarations(self, ctx:sclParser.BlockConstDeclarationsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#blockStaticDeclarations.
    def visitBlockStaticDeclarations(self, ctx:sclParser.BlockStaticDeclarationsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#blockConstantDeclarations.
    def visitBlockConstantDeclarations(self, ctx:sclParser.BlockConstantDeclarationsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#blockLabelDeclarations.
    def visitBlockLabelDeclarations(self, ctx:sclParser.BlockLabelDeclarationsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#blockFunctionality.
    def visitBlockFunctionality(self, ctx:sclParser.BlockFunctionalityContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#function_type.
    def visitFunction_type(self, ctx:sclParser.Function_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#stat.
    def visitStat(self, ctx:sclParser.StatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#functionStatement.
    def visitFunctionStatement(self, ctx:sclParser.FunctionStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#labelStatment.
    def visitLabelStatment(self, ctx:sclParser.LabelStatmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#forInitialCondition.
    def visitForInitialCondition(self, ctx:sclParser.ForInitialConditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#forEndCondition.
    def visitForEndCondition(self, ctx:sclParser.ForEndConditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#forStepCondition.
    def visitForStepCondition(self, ctx:sclParser.ForStepConditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#forBlockStatements.
    def visitForBlockStatements(self, ctx:sclParser.ForBlockStatementsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#ifCondition.
    def visitIfCondition(self, ctx:sclParser.IfConditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#ifBlockStatments.
    def visitIfBlockStatments(self, ctx:sclParser.IfBlockStatmentsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#switchBlock.
    def visitSwitchBlock(self, ctx:sclParser.SwitchBlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#switchLabel.
    def visitSwitchLabel(self, ctx:sclParser.SwitchLabelContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#switchLabelConstant.
    def visitSwitchLabelConstant(self, ctx:sclParser.SwitchLabelConstantContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#switchBlockStatements.
    def visitSwitchBlockStatements(self, ctx:sclParser.SwitchBlockStatementsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#defualtswitchBlockStatements.
    def visitDefualtswitchBlockStatements(self, ctx:sclParser.DefualtswitchBlockStatementsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#assignmentStatement.
    def visitAssignmentStatement(self, ctx:sclParser.AssignmentStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#leftHandAssignment.
    def visitLeftHandAssignment(self, ctx:sclParser.LeftHandAssignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#rightHandAssignment.
    def visitRightHandAssignment(self, ctx:sclParser.RightHandAssignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#expr.
    def visitExpr(self, ctx:sclParser.ExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#varName.
    def visitVarName(self, ctx:sclParser.VarNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#functionParameterExpression.
    def visitFunctionParameterExpression(self, ctx:sclParser.FunctionParameterExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#inputParameter.
    def visitInputParameter(self, ctx:sclParser.InputParameterContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#outputParameter.
    def visitOutputParameter(self, ctx:sclParser.OutputParameterContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#numberConstant.
    def visitNumberConstant(self, ctx:sclParser.NumberConstantContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#constant.
    def visitConstant(self, ctx:sclParser.ConstantContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#expressionName.
    def visitExpressionName(self, ctx:sclParser.ExpressionNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#ambiguousName.
    def visitAmbiguousName(self, ctx:sclParser.AmbiguousNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#ambiguousNext.
    def visitAmbiguousNext(self, ctx:sclParser.AmbiguousNextContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#identifierIndex.
    def visitIdentifierIndex(self, ctx:sclParser.IdentifierIndexContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by sclParser#identifierIndexValue.
    def visitIdentifierIndexValue(self, ctx:sclParser.IdentifierIndexValueContext):
        return self.visitChildren(ctx)



del sclParser